(define (problem picture_86)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_pump red_pump green_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_pump)
        (clear red_pump)
        (clear green_pump)
        (part_at blue_pump table)
        (part_at green_pump table)
        (part_at red_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
        )
    )
)
