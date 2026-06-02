(define (problem picture_41)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery green_regulator red_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_regulator)
        (on blue_battery red_battery)
        (part_at green_regulator table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at green_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
