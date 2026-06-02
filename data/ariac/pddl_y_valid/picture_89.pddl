(define (problem picture_89)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery red_battery green_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear red_battery)
        (clear green_battery)
        (part_at blue_battery table)
        (part_at green_battery table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at blue_battery battery_placement)
        )
    )
)
