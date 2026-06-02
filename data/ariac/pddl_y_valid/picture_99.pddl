(define (problem picture_99)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_pump blue_battery green_regulator red_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_pump)
        (clear blue_battery)
        (clear red_pump)
        (part_at red_pump table)
        (part_at blue_pump pump_placement)
        (part_at blue_battery battery_placement)
        (part_at green_regulator regulator_placement)
        (clear green_regulator)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
        )
    )
)
